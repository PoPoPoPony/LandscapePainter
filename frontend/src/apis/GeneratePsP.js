// import request from '../../../utils/requestBackend'
import request from '@/utils/requestBackend'


export function GeneratePsP(rgbArray) {
    var data = JSON.stringify(rgbArray)
    let config = {
        url: '/GeneratePsP',
        method: 'POST',
        headers: {
            "Content-Type": "application/json"
        },
        data: data
    }
    

    return request(config)
}